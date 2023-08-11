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
	public:
		cacheCTR(const std::string& config_file);

		unsigned int current_line_ctr_num; //计数值

		map<long, long> ctr_value;

		INIReader* reader_cacheCTR;
		void InitCache()override;
		void GetHitNum(char *address, dramsim3::CPU *cpu)override;
		void PrintCache(std::string file_name)override;
		void AddMiss2Entry(bitset<32> flags, int acc, dramsim3::CPU *cpu)override;

	};
}
#endif