#ifndef _CACHEMAC_H_
#define _CACHEMAC_H_

#include"cache.h"

namespace dramsim3 {

	//MAC
	class cacheMAC :public Cache {
	private:
		char* address;
		bool get_next = true;
		std::string addr, operation, added_cycle, type;


	public:
		INIReader* reader_cacheMAC;

		cacheMAC(const std::string& config_file);
		void InitCache()override;
		void GetWrite(CPU *cpu)override;
		void cache_clock(dramsim3::CPU *cpu)override;
		void PrintCache(string layer_name, string index_layer_, string file)override;
		void AddMiss2Entry(bitset<32> flags, int acc, dramsim3::CPU *cpu)override;
		int mac_length;
		int hash_delay = 0;


		bool IsHit(bitset<32> flags)override;
		void LruHitProcess()override;
		void LruUnhitSpace() override;
		void LruUnhitUnspace()override;
		void GetRead(bitset<32> flags, CPU *cpu)override;
		void GetReplace(bitset<32> flags, CPU *cpu) override;

		void MSHR_cycle(bitset<32> flags, dramsim3::CPU *cpu)  override;
		std::pair<bool, bool> WillAcceptMiss(bitset<32> flags)  override;
		void ReleaseMSHR(dramsim3::CPU *cpu)  override;
	};
}
#endif
