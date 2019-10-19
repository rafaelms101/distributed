#ifndef SYNCBUFFER_H_
#define SYNCBUFFER_H_

#include <mutex> 
#include <condition_variable>
#include <atomic>

#include "utils.h"

class SyncBuffer {
public:
	SyncBuffer(const long element_size_in_bytes, const long max_entries);
	virtual ~SyncBuffer() {}
	
	double arrivalInterval();
	void insert(const long num_elements, byte* data);
	void remove(const long num_elements);
	byte* front();
	byte* fromElement(const long element);
	void waitForData(const long num_elements);
	long num_entries();
	
private:
	byte* data = nullptr;
	long start = 0;
	long end = 0;
	std::atomic<long> num_elements_stored;
	double last_insert = 0;
	long element_size_in_bytes;
	long total_size_in_bytes;
	
	std::mutex mutex;
	std::condition_variable cv;
	bool waiting = false;
	long waiting_quantity = 0;
};

#endif /* SYNCBUFFER_H_ */
