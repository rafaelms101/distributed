#ifndef CIRCULARBUFFER_H_
#define CIRCULARBUFFER_H_

#include <stdexcept>
#include <assert.h>
#include <mutex>   
#include <shared_mutex>
#include <condition_variable>


class QueryBuffer {
public:
	double block_rate();
	QueryBuffer(const long block_size, const long num_entries);
	~QueryBuffer();
	long bs();
	bool hasSpace();
	void add();
	unsigned char* peekFront();
	unsigned char* peekEnd();
	void consume(int n);
	bool empty();
	void waitForData(int n);
	int entries();
	
private:
	bool waiting = false;
	long waiting_min_data = 0;

	unsigned char* data;
	long start = 0;
	long end = 0;
	long used = 0;
	
	double br = 0;
	double last_block_time;
	
	const long block_size;
	const long total_size;
	std::shared_mutex mutex;
	std::condition_variable_any has_data;
};

#endif
