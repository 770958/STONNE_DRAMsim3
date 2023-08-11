#ifndef _CACHEBMT_H_
#define _CACHEBMT_H_

#include"cache.h"

namespace dramsim3 {

	//BMT
	class cacheBMT :public Cache {
	public:
		INIReader* reader_cacheBMT;

		cacheBMT(const std::string& config_file);
		void InitCache()override;
		void GetHitNum(char *address, dramsim3::CPU *cpu)override;
		void PrintCache(std::string file_name)override;

	};
}
#endif