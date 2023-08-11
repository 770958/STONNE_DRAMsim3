#ifndef _CACHEMAC_H_
#define _CACHEMAC_H_

#include"cache.h"

namespace dramsim3 {

	//MAC
	class cacheMAC :public Cache {
	public:
		INIReader* reader_cacheMAC;

		cacheMAC(const std::string& config_file);
		void InitCache()override;
		void GetHitNum(char *address, dramsim3::CPU *cpu)override;
		void PrintCache(std::string file_name)override;

	};
}
#endif