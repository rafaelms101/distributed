#ifndef CIRCULARBUFFER_H_
#define CIRCULARBUFFER_H_

#include <stdexcept>
#include <assert.h>
#include <mutex>   
#include <condition_variable>


class Buffer {
public:
	Buffer(const long block_size, const long num_entries);
	~Buffer();

	double block_rate();
	bool hasSpace(const long qty = 1);
	void add(const long qty = 1);
	unsigned char* peekFront();
	unsigned char* peekEnd();
	void consume(const long n);
	void waitForData(int n);
	void waitForSpace(int n);
	int entries();
	
private:
	void maybe_realign();
	
	long used() { return end - start; }
	long available() { return total_size - end; }
	
	long waiting_to_produce = 0;
	long waiting_to_consume = 0;

	unsigned char* data;
	long start = 0;
	long end = 0;
	
	double time_between_blocks = 0;
	double last_block_time = 0;
	
	const long block_size;
	const long total_size;
	
	std::mutex mutex;
	std::condition_variable can_consume;
	std::condition_variable can_produce;
};

#endif
