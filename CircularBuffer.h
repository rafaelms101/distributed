#ifndef CIRCULARBUFFER_H_
#define CIRCULARBUFFER_H_

#include <stdexcept>
#include <assert.h>
#include <mutex>   
#include <shared_mutex>

class SimpleBuffer {
public:
	SimpleBuffer(const long block_size, const long num_entries) {
		data = new unsigned char[block_size * num_entries];
		end = 0;
	}
	
	unsigned char* data;
	long end;
};

//TODO: This is just something temporary for the sake of testing
class Buffer {
public:
	Buffer(const long block_size, const int num_entries) : block_size{block_size}, total_size{block_size * num_entries} {
		data = new unsigned char[total_size];
	}
	
	~Buffer() {
		delete [] data;
	}
	
	
	long bs() {
		std::shared_lock {mutex};
		return block_size;
	}
	
	bool hasSpace() {
		std::shared_lock {mutex};
		return total_size - end >= block_size;
	}
	
	unsigned char* next() {
		std::unique_lock {mutex};
		
		num_entries++;
		
		if (total_size - end >= block_size) {
			//adding to the front
			end += block_size;
			return &data[end - block_size];
		} else {
			throw std::out_of_range("trying to add an element to a full buffer");
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
	
	void removeFront() {
		std::unique_lock {mutex};
		
		num_entries--;
		start += block_size;
		assert(start <= end);
	}
	
	bool empty() {
		std::shared_lock {mutex};
		
		return start == end;
	}
	
	int entries() {
		std::shared_lock {mutex};
		
		return num_entries;
	}
	 
private:
	std::shared_mutex mutex;
	unsigned char* data;
	long start = 0;
	long end = 0;
	int num_entries = 0;
	
	const long block_size;
	const long total_size;
};

class CircularBuffer {
	//TODO: add peekEnd
public:
	CircularBuffer(const long block_size, const int num_entries) : block_size{block_size}, total_size{block_size * num_entries} {
		data = new unsigned char[total_size];
	}
	
	~CircularBuffer() {
		delete [] data;
	}
	
	
	long bs() {
		std::shared_lock {mutex};
		return block_size;
	}
	
	bool hasSpace() {
		std::shared_lock {mutex};
		
		return total_size - end1 >= block_size || start - end2 >= block_size;
	}
	
	unsigned char* next() {
		std::unique_lock {mutex};
		
		num_entries++;
		
		if (total_size - end1 >= block_size) {
			//adding to the front
			end1 += block_size;
			return &data[end1 - block_size];
		} else if (start - end2 >= block_size) {
			//adding to the back
			end2 += block_size;
			return &data[end2 - block_size];
		} else {
			throw std::out_of_range("trying to add an element to a full buffer");
		}
	}
	
	unsigned char* peekFront() {
		std::shared_lock {mutex};
		
		return &data[start];
	}
	
	void removeFront() {
		std::unique_lock {mutex};
		
		num_entries--;
		start += block_size;
		assert(start <= end1);
		
		if (start == total_size) {
			start = 0;
			end1 = end2;
			end2 = 0;
		}
	}
	
	bool empty() {
		std::shared_lock {mutex};
		
		return start == end1;
	}
	
	int entries() {
		std::shared_lock {mutex};
		return num_entries;
	}
	 
private:
	unsigned char* data;
	long start = 0;
	long end1 = 0;
	long end2 = 0;
	int num_entries = 0;
	
	const long block_size;
	const long total_size;
	std::shared_mutex mutex;
};

#endif
