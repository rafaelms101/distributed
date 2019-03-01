#ifndef CIRCULARBUFFER_H_
#define CIRCULARBUFFER_H_

#include <stdexcept>
#include <assert.h>
#include <mutex>   
#include <shared_mutex>
//
////TODO: This is just something temporary for the sake of testing
//class Buffer {
//public:
//	Buffer(const long block_size, const int num_entries) : block_size{block_size}, total_size{block_size * num_entries} {
//		data = new unsigned char[total_size];
//	}
//	
//	~Buffer() {
//		delete [] data;
//	}
//	
//	
//	long bs() {
//		std::shared_lock {mutex};
//		return block_size;
//	}
//	
//	bool hasSpace() {
//		std::shared_lock {mutex};
//		return total_size - end >= block_size;
//	}
//	
//	unsigned char* next() {
//		std::unique_lock {mutex};
//		
//		num_entries++;
//		
//		if (total_size - end >= block_size) {
//			//adding to the front
//			end += block_size;
//			return &data[end - block_size];
//		} else {
//			throw std::out_of_range("trying to add an element to a full buffer");
//		}
//	}
//	
//	unsigned char* peekFront() {
//		std::shared_lock {mutex};
//		
//		return &data[start];
//	}
//	
//	unsigned char* peekEnd() {
//		std::shared_lock {mutex};
//		
//		return &data[end];
//	}
//	
//	void removeFront() {
//		std::unique_lock {mutex};
//		
//		num_entries--;
//		start += block_size;
//		assert(start <= end);
//	}
//	
//	bool empty() {
//		std::shared_lock {mutex};
//		
//		return start == end;
//	}
//	
//	int entries() {
//		std::shared_lock {mutex};
//		
//		return num_entries;
//	}
//	 
//private:
//	std::shared_mutex mutex;
//	unsigned char* data;
//	long start = 0;
//	long end = 0;
//	int num_entries = 0;
//	
//	const long block_size;
//	const long total_size;
//};

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
	 
private:
	unsigned char* data;
	long start = 0;
	long end = 0;
	long used = 0;
	
	const long block_size;
	const long total_size;
	std::shared_mutex mutex;
};

#endif
