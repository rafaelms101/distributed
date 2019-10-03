#include "ExecPolicy.h"

int BenchExecPolicy::numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg) {
	if (nrepeats >= BENCH_REPEATS) {
		nrepeats = 0;
		nb++;
	}

	nrepeats++;
	return nb;
}

std::pair<int, int> DynamicExecPolicy::longest_contiguous_region(double min, double tolerance, std::vector<double>& time_per_block) {
	int start, bestStart, bestEnd;
	int bestLength = 0;
	int length = 0;

	double threshold = min * (1 + tolerance);

	for (int i = 1; i < time_per_block.size(); i++) {
		if (time_per_block[i] <= threshold) {
			length++;

			if (length > bestLength) {
				bestStart = start;
				bestEnd = i;
				bestLength = length;
			}
		} else {
			start = i + 1;
			length = 0;
		}
	}

	return std::pair<int, int>(bestStart, bestEnd);
}

void DynamicExecPolicy::setup() {
	std::vector<double> times(load_prof_times(true, cfg));
	std::vector<double> time_per_block(times.size());

	for (int i = 1; i < times.size(); i++) {
		time_per_block[i] = times[i] / i;
	}

	double tolerance = 0.1;
	int minBlock = 1;

	for (int nb = 1; nb < times.size(); nb++) {
		if (time_per_block[nb] < time_per_block[minBlock]) minBlock = nb;
	}

	std::pair<int, int> limits = longest_contiguous_region(time_per_block[minBlock], 0.1, time_per_block);
	pdGPU.min_block = limits.first;
	pdGPU.max_block = limits.second;

	pdGPU.times = new double[pdGPU.min_block + 1];
	pdGPU.times[0] = 0;
	for (int nb = 1; nb <= pdGPU.min_block; nb++)
		pdGPU.times[nb] = times[nb];

	deb("min=%d, max=%d", pdGPU.min_block * cfg.block_size, pdGPU.max_block * cfg.block_size);
	//	std::printf("min=%d\n", pd.min_block * cfg.block_size);
	assert(pdGPU.max_block <= cfg.eval_length);
}

int MinExecPolicy::numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);

	int num_blocks = buffer.entries();

	if (num_blocks < pdGPU.min_block) {
		double work_without_waiting = num_blocks / pdGPU.times[num_blocks];
		double work_with_waiting = pdGPU.min_block / ((pdGPU.min_block - num_blocks) * buffer.block_interval() + pdGPU.times[pdGPU.min_block]);

		if (work_with_waiting > work_without_waiting) {
			usleep((pdGPU.min_block - num_blocks) * buffer.block_interval() * 1000000);
		}
	}

	return std::min(buffer.entries(), pdGPU.min_block);
}

int MaxExecPolicy::numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);

	int num_blocks = buffer.entries();

	if (num_blocks < pdGPU.min_block) {
		double work_without_waiting = num_blocks / pdGPU.times[num_blocks];
		double work_with_waiting = pdGPU.min_block / ((pdGPU.min_block - num_blocks) * buffer.block_interval() + pdGPU.times[pdGPU.min_block]);

		if (work_with_waiting > work_without_waiting) {
			usleep((pdGPU.min_block - num_blocks) * buffer.block_interval() * 1000000);
		}
	}

	return std::min(buffer.entries(), pdGPU.max_block);
}	

int QueueExecPolicy::numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);

	while (true) {
		//deciding between executing what we have or wait one extra block
		int num_blocks = buffer.entries();
		int nq = num_blocks * cfg.block_size;

		if (num_blocks >= pdGPU.min_block) {
			processed += pdGPU.min_block * cfg.block_size;
			return pdGPU.min_block;
		}

		if (nq + processed == cfg.test_length) return num_blocks;

		//case 1: execute right now
		int queries_after_execute = pdGPU.times[num_blocks] / buffer.block_interval();

		//case 2: wait for one more block
		int queries_after_wait = pdGPU.times[num_blocks + 1] / buffer.block_interval();

		if (queries_after_execute <= nq) {
			processed += nq;
			return num_blocks;
		}

		if (queries_after_wait <= nq) {
			buffer.waitForData(num_blocks + 1);
			continue;
		}

		double increase_rate_execute = double(queries_after_execute - nq) / pdGPU.times[num_blocks];
		double increase_rate_wait = double(queries_after_wait - nq) / (buffer.block_interval() + pdGPU.times[num_blocks + 1]);

		if (increase_rate_execute <= increase_rate_wait) {
			processed += nq;
			return num_blocks;
		}

		buffer.waitForData(num_blocks + 1);
	}
}

int QueueMaxExecPolicy::numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);

	while (true) {
		//deciding between executing what we have or wait one extra block
		int num_blocks = buffer.entries();
//		if (processed % 1000 == 0) std::printf("processed: %d\n", processed);
		int nq = num_blocks * cfg.block_size;

		if (num_blocks >= pdGPU.max_block) {
			processed += pdGPU.max_block * cfg.block_size;
			return pdGPU.max_block;
		}

		if (nq + processed == cfg.test_length) return num_blocks;

		//case 1: execute right now
		int queries_after_execute = pdGPU.times[num_blocks] / buffer.block_interval();

		//case 2: wait for one more block
		int queries_after_wait = pdGPU.times[num_blocks + 1] / buffer.block_interval();

		if (queries_after_execute <= nq) {
			processed += nq;
			return num_blocks;
		}

		if (queries_after_wait <= nq) {
			buffer.waitForData(num_blocks + 1);
			continue;
		}

		double increase_rate_execute = double(queries_after_execute - nq) / pdGPU.times[num_blocks];
		double increase_rate_wait = double(queries_after_wait - nq) / (buffer.block_interval() + pdGPU.times[num_blocks + 1]);

		if (increase_rate_execute <= increase_rate_wait) {
			processed += nq;
			return num_blocks;
		}

		buffer.waitForData(num_blocks + 1);
	}
}

int MinGreedyExecPolicy::numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);
	int num_blocks = buffer.entries();
	return std::min(num_blocks, pdGPU.min_block);
}


