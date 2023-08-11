#ifndef _CACHEL2_H_
#define _CACHEL2_H_

#include"cache.h"

namespace dramsim3 {

	//L2
	class cacheL2 :public Cache {
	private:
		char* address;
		bool get_next = true;
		//cacheFifo* this_l2_fifo_rtn;
		std::string addr, operation, added_cycle, type;

	public:
		INIReader* reader_cacheL2;

		cacheL2(const std::string& config_file);
		void InitCache()override;
		void GetWrite(CPU *cpu)override;
		void cache_clock(dramsim3::CPU *cpu)override;
		void PrintCache(string layer_name, string index_layer_, string file)override;
		//void send(dramsim3::CPU *cpu);
		int complete = 0;
		int fifo_size = 0;
		bool IsHit(bitset<32> flags)override;
		void LruHitProcess()override;
		void LruUnhitSpace()override;
		void LruUnhitUnspace()override;
		void GetRead(bitset<32> flags, CPU *cpu)override;
		void GetReplace(bitset<32> flags, CPU *cpu)override;

		void MSHR_cycle(bitset<32> flags, dramsim3::CPU *cpu)override;
		std::pair<bool, bool> WillAcceptMiss(bitset<32> flags)override;
		void AddMiss2Entry(bitset<32> flags, int acc, dramsim3::CPU *cpu)override;
		void ReleaseMSHR(dramsim3::CPU *cpu)override;
	};
}


#endif
