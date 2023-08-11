#include <iostream>
#include "./../ext/headers/args.hxx"
#include "cpu.h"

using namespace dramsim3;

int main(int argc, const char **argv) {
	args::ArgumentParser parser(
		"DRAM Simulator.",
		"Examples: \n."
		"./build/dramsim3main configs/DDR4_8Gb_x8_3200.ini"
		"-t sample_trace.txt  -o out.txt\n");
	args::HelpFlag help(parser, "help", "Display the help menu", { 'h', "help" });
	args::ValueFlag<std::string> output_dir_arg(
		parser, "output_dir", "Output directory for stats files",
		{ 'o', "output-dir" }, ".");
	args::ValueFlag<std::string> trace_file_arg(
		parser, "trace", "Trace file",
		{ 't', "trace" });
	args::Positional<std::string> config_arg(
		parser, "config", "The config file name (mandatory)");

	try {
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help) {
		std::cout << parser;
		return 0;
	}
	catch (args::ParseError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

	std::string config_file = args::get(config_arg);
	if (config_file.empty()) {
		std::cerr << parser;
		return 1;
	}

 	std::string output_dir = args::get(output_dir_arg);
	std::string trace_file = args::get(trace_file_arg);
 
	std::string command = "mkdir -p " + output_dir;
	system(command.c_str());

	CPU *cpu;
	cpu = new TraceBasedCPU(config_file, output_dir);
	cpu->setTracefile(trace_file);

	bool cpu_finished = false;

	while (!cpu_finished || !cpu->l2_fifo->isEmpty() || !cpu->dram_fifo->isEmpty()) {
		//std::cout <<"origin£º "<< cpu->dram_fifo->size() << std::endl;

		if (!cpu_finished) {
			cpu->setAccessConfig();
		}
		

		cpu->l2->cache_clock(cpu);
		cpu->mac_cache->cache_clock(cpu);
		cpu->ctr_cache->cache_clock(cpu);
		cpu->bmt_cache->cache_clock(cpu);
		cpu->AccessDram();

		cpu->partition_cycle++;
		cpu_finished = cpu->isFinished();

		//std::cout << "now£º " << cpu->dram_fifo->size() << std::endl;
	}
	cpu->PrintStats();
	cpu->GetDelay();
	cpu->l2->PrintCache("CONV", "0001", output_dir);
	cpu->ctr_cache->PrintCache("CONV", "0001", output_dir);
	cpu->bmt_cache->PrintCache("CONV", "0001", output_dir);
	cpu->mac_cache->PrintCache("CONV", "0001", output_dir);
	
	string cpu_out_file = output_dir + "/delay.txt";
	ofstream ofs;
	ofs.open(cpu_out_file, ios::out | ios::app);
	ofs << "total_delay" << "," << "avg_delay" << "," << "total_num" << "," << "partition_cycle" << endl;
	ofs << cpu->total_read_delay << "," << cpu->avg_read_delay << "," << cpu->total_read_num << "," << cpu->partition_cycle << endl;
	ofs.close();


	delete cpu;

	return 0;
}
