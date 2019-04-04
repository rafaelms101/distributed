#ifndef CIRCULARBUFFER_H_
#define CIRCULARBUFFER_H_

#include <stdexcept>
#include <assert.h>
#include <mutex>   
#include <shared_mutex>
#include <condition_variable>

class CircularBuffer {
	//TODO: add peekEnd
public:
	CircularBuffer(const long block_size, const long num_entries) : block_size{block_size}, total_size{block_size * num_entries} {
		data = new unsigned char[total_size];
	}
	
	~CircularBuffer() {
		delete [] data;
	}
	
	
	long bs() {
		return block_size;
	}
	
	bool hasSpace() {
		std::shared_lock {mutex};
		return used < total_size;
	}
	
	void add() {
		std::unique_lock {mutex};
		
		used += block_size;
		end = (end + block_size) % total_size;
		
		if (waiting && used >= waiting_min_data) {
			waiting = false;
			has_data.notify_one();
		}
	}
	
	unsigned char* peekFront() {
		std::shared_lock {mutex};
		return &data[start];
	}
	
	unsigned char* peekEnd() {
		std::shared_lock {mutex};
		return &data[end];
	}
	
	void consume(int n) {
		std::unique_lock {mutex};
		used -= n * block_size;
		start = (start + n * block_size) % total_size;
	}
	
	bool empty() {
		std::shared_lock {mutex};		
		return used == 0;
	}
	
	long entries() {
		std::shared_lock {mutex};	
		return used / block_size;
	}
	
	void waitForData(int n) {
		std::unique_lock lck {mutex};
		
		waiting_min_data = n * block_size;

		if (used < waiting_min_data) {
			waiting = true;
			has_data.wait(lck, [this] {return ! waiting;});
		}
	}
	
	
	 
private:
	bool waiting = false;
	long waiting_min_data;

	unsigned char* data;
	long start = 0;
	long end = 0;
	long used = 0;
	
	const long block_size;
	const long total_size;
	std::shared_mutex mutex;
	std::condition_variable_any has_data;
};

#endif
