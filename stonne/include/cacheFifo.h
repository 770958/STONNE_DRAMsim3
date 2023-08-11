

#ifndef __cacheFifo_h__
#define __cacheFifo_h__

#include <queue>
#include "DataPackage.h"
#include "types.h"
#include "Stats.h"

namespace dramsim3 {
	class cacheFifo {
	private:
		std::queue<int> fifo;
		unsigned int capacity; //Capacity in number of bits
		unsigned int capacity_words; //Capacity in number of words allowed. i.e., capacity_words = capacity / size_word
		FifoStats fifoStats; //Tracking parameters
	public:
		cacheFifo( int capacity);
		bool isEmpty();
		bool isFull();
		void push(int num);
		int pop();
		int front();
 		unsigned int size(); //Return the number of elements in the fifo 
	};
}

#endif
