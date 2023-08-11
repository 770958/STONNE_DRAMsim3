#ifndef _CACHECTR_H_
#define _CACHECTR_H_

#include"cache.h"
//#include"cpu.h"
#include <map>
#include<string>
namespace dramsim3 {

	class CPU;	//前向声明
	//counter
	class cacheCTR :public Cache {
	private:
		char* address;
		bool get_next = true;
		std::string addr, operation, added_cycle, type;
	public:
		cacheCTR(const std::string& config_file);

		unsigned int current_line_ctr_num; //计数值

		map<long, long> ctr_value;

		INIReader* reader_cacheCTR;
		void InitCache()override;
		void GetWrite(CPU *cpu)override;
		void cache_clock(dramsim3::CPU *cpu)override;
		void PrintCache(string layer_name, string index_layer_, string file)override;
		void AddMiss2Entry(bitset<32> flags, int acc, dramsim3::CPU *cpu)override;
		//void send(dramsim3::CPU *cpu);
		int encry_delay = 0;


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
