#ifndef PQUEUE_H_
#define PQUEUE_H_

#include <utility>
#include <algorithm>
#include <assert.h>


bool cmpPair(const std::pair<float, long> &a, const std::pair<float, long> &b) {
	return a.first < b.first;
}

struct pqueue {
	pqueue(std::pair<float, long>* _base, int _max) {
		size = 0;
		max = _max;
		base = _base;
	}
	
	void add(float dist, long id) {
		if (size < max) {
			base[size] = std::make_pair(dist, id);
			size++;
			
			if (size == max) {
				std::make_heap(base, base + max, cmpPair);
			}
		}
		else if (dist < base[0].first) {
			std::pop_heap(base, base + max, cmpPair);
			base[max-1] = std::make_pair(dist, id);
			std::push_heap(base, base + max - 1, cmpPair);
		}
	}
	
	std::pair<float, long> pop() {
		assert(size != 0);
		
		size -= 1;
		std::pop_heap(base, base + size, cmpPair);
		
		return base[size];
	}

	int size;
	int max;
	std::pair<float, long>* base;
};


#endif /* PQUEUE_H_ */
