#include "QueryBuffer.h"

#include <sys/time.h>
#include <cstring>

static double now() {
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

double QueryBuffer::block_rate() {
	std::shared_lock { mutex };
	return br;
}

QueryBuffer::QueryBuffer(const long block_size, const long num_entries) : block_size { block_size }, total_size { block_size * num_entries } {
	data = new unsigned char[total_size];
	last_block_time = now();
}

QueryBuffer::~QueryBuffer() {
	delete[] data;
}

long QueryBuffer::bs() {
	return block_size;
}

bool QueryBuffer::hasSpace() {
	std::unique_lock { mutex };
	
	if (end == total_size && start >= total_size / 2) realign();
	
	return end < total_size;
}

inline void QueryBuffer::realign() {
	std::printf("REALIGN: %d queries\n", used / block_size * 20);
	//move to the beginning
	std::memmove(data, data + start, used);
	start = 0;
	end = used;
}

void QueryBuffer::add() {
	std::unique_lock { mutex };

	double n = now();
	br = n - last_block_time;
	last_block_time = n;
	
	used += block_size;
	end = end + block_size;
	
	

	if (waiting && used >= waiting_min_data) {
		waiting = false;
		has_data.notify_one();
	}
}

unsigned char* QueryBuffer::peekFront() {
	std::shared_lock { mutex };
	return &data[start];
}

unsigned char* QueryBuffer::peekEnd() {
	std::shared_lock { mutex };
	
	return &data[end];
}

void QueryBuffer::consume(int n) {
	std::unique_lock { mutex };
	used -= n * block_size;
	start = start + n * block_size;
}

bool QueryBuffer::empty() {
	std::shared_lock { mutex };
	return used == 0;
}

int QueryBuffer::entries() {
	std::shared_lock { mutex };
	return used / block_size;
}

void QueryBuffer::waitForData(int n) {
	std::unique_lock lck { mutex };

	waiting_min_data = n * block_size;

	if (used < waiting_min_data) {
		waiting = true;
		has_data.wait(lck, [this] {return ! waiting;});
	}
}
