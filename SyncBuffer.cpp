#include "SyncBuffer.h"

#include <cstring>

SyncBuffer::SyncBuffer(const long _element_size_in_bytes, const long max_entries) {
	num_elements_stored = 0;
	element_size_in_bytes = _element_size_in_bytes;
	total_size_in_bytes = element_size_in_bytes * max_entries;
	data = new byte[total_size_in_bytes];
}

double SyncBuffer::arrivalInterval() {
	return now() - last_insert;
}

void SyncBuffer::insert(const long num_elements, byte* const new_data) {
	mutex.lock();
	
	last_insert = now();
	
	long data_size = num_elements * element_size_in_bytes;
	
	assert(data_size + end <= total_size_in_bytes);
	
	std::memcpy(&data[end], new_data, data_size);
	end += data_size;
	num_elements_stored += num_elements;
	
	if (waiting && num_elements_stored >= waiting_quantity) {
		waiting = false;
		cv.notify_one();
	} 
	
	mutex.unlock();
}

void SyncBuffer::remove(const long num_elements) {
	long data_size = num_elements * element_size_in_bytes;
	start += data_size;
	num_elements_stored -= num_elements;
}

byte* SyncBuffer::front() {
	return &data[start];
}

byte* SyncBuffer::fromElement(const long element) {
	return &data[start + element * element_size_in_bytes];
}

void SyncBuffer::waitForData(const long num_elements) {
	std::unique_lock<std::mutex> lck { mutex };

	if (num_elements_stored < num_elements) {
		waiting_quantity = num_elements;
		waiting = true;
		cv.wait(lck, [this] { return ! waiting; });
	}
}

long SyncBuffer::num_entries() {
	return num_elements_stored;
}
