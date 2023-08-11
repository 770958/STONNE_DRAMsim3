//created by wangjs
#include "cacheFifo.h"
#include <assert.h>
#include <iostream>
#include "utility.h"
namespace dramsim3 {
	cacheFifo::cacheFifo( int capacity) {
		this->capacity = capacity;
		this->capacity_words = capacity / sizeof(data_t); //Data size
	}

	bool cacheFifo::isFull() {
		return  this->fifo.size() >= this->capacity_words;  // > is forbidden
	}

	bool cacheFifo::isEmpty() {
		return this->fifo.size() == 0;
	}

	void cacheFifo::push(int num) {
		//    assert(!isFull());  //The fifo must not be full
		fifo.push(num); //Inserting at the end of the queue
		if (this->size() > this->fifoStats.max_occupancy) {
			this->fifoStats.max_occupancy = this->size();
		}
		this->fifoStats.n_pushes += 1; // To track information

	}

	int cacheFifo::pop() {
		assert(!isEmpty());
		this->fifoStats.n_pops += 1; //To track information
		int pck = fifo.front() ; //Accessing the first element of the queue
		fifo.pop(); //Extracting the first element
		return pck;
	}

	int cacheFifo::front() {
		assert(!isEmpty());
		int pck = fifo.front() ;
		this->fifoStats.n_fronts += 1; //To track information
		return pck;
	} 

	unsigned int cacheFifo::size() {
		return fifo.size();
	}

}