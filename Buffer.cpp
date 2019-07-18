#include "Buffer.h"

#include <sys/time.h>
#include <cstring>

static double now() {
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

Buffer::Buffer(const long block_size, const long num_entries) : block_size { block_size }, total_size { block_size * num_entries } {
	data = new unsigned char[total_size];
	last_block_time = now();
}

Buffer::~Buffer() {
	delete[] data;
}

double Buffer::block_interval() {
	std::unique_lock<std::mutex> { mutex };
	return time_between_blocks;
}

inline void Buffer::maybe_realign() {
	if (start >= 0.4 * total_size) {
		assert(false);
		//move to the beginning
		std::memmove(data, data + start, used());
		end = used();
		start = 0;
	}
}

void Buffer::add(const long qty) {
	std::unique_lock<std::mutex> { mutex };

	//TODO: this should be outside of this class
	double n = now();
	time_between_blocks = n - last_block_time;
	last_block_time = n;

	end = end + block_size * qty;
	
	if (waiting_to_consume > 0 && used() >= waiting_to_consume) {
		waiting_to_consume = 0;
		can_consume.notify_one();
	}
}

unsigned char* Buffer::peekFront() {
	std::unique_lock<std::mutex> { mutex };
	return &data[start];
}

unsigned char* Buffer::peekEnd() {
	std::unique_lock<std::mutex> { mutex };
	
	return &data[end];
}

void Buffer::consume(const long n) {
	std::unique_lock<std::mutex> { mutex };

	start = start + n * block_size;

	if (waiting_to_produce > 0) {
		maybe_realign();
		
		if (available() >= waiting_to_produce) {
			waiting_to_produce = 0;
			can_produce.notify_one();
		}
	}
}

int Buffer::entries() {
	std::unique_lock<std::mutex> { mutex };
	return used() / block_size;
}

void Buffer::waitForSpace(int n) {
	std::unique_lock<std::mutex> lck { mutex };

	waiting_to_produce = n * block_size;

	if (available() < waiting_to_produce) {
		maybe_realign();
		
		if (available() < waiting_to_produce) {
			can_produce.wait(lck, [this] { return waiting_to_produce == 0; });
		}
	} else {
		waiting_to_produce = 0;
	}
}

void Buffer::waitForData(int n) {
	std::unique_lock<std::mutex> lck { mutex };

	waiting_to_consume = n * block_size;

	if (used() < waiting_to_consume) {
		can_consume.wait(lck, [this] { return waiting_to_consume == 0; });
	} else {
		waiting_to_consume = 0;
	}
}

void Buffer::transfer(void* ptr, int num_blocks) {
	waitForSpace(num_blocks);
	std::memcpy(data + end, ptr, block_size * num_blocks);
	add(num_blocks);
}


