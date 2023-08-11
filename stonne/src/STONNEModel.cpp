//
// Created by Francisco Munoz-Martinez on 18/06/19.
//
#include "STONNEModel.h"

#include <assert.h>
#include <chrono>
#include "types.h"
#include <vector>
#include "Tile.h"
#include "utility.h"
#include "Config.h"
#include <time.h>
#include <Stats.h>
#include <algorithm> 
#include <stdlib.h>
#include <string>
#include <sstream>
#include<iomanip>

using namespace std;
using namespace dramsim3;

// added by wangjs
int index_layer = 1;
const string output_dir_ = "../dramsim3/output1/";
string config_file = "../dramsim3/configs/ini/DDR4_8Gb_x8_3200.ini";
 
unsigned long long global_total_read_delay = 0;
double global_avg_read_delay = 0.00;
unsigned long long global_total_read_num = 0;
unsigned long long global_n_cycles = 0;
// added end


Stonne::Stonne(Config_stonne stonne_cfg) {
    this->stonne_cfg=stonne_cfg;
	this->model_name = stonne_cfg.model_name;
	this->config_file_ = stonne_cfg.config_file;
	this->arch_name = stonne_cfg.arch_name;
	this->ms_size = stonne_cfg.m_MSNetworkCfg.ms_size;
    this->layer_loaded=false;
    this->tile_loaded=false;
    this->outputASConnection = new Connection(stonne_cfg.m_SDMemoryCfg.port_width);
    this->outputLTConnection = new Connection(stonne_cfg.m_LookUpTableCfg.port_width);
    switch(stonne_cfg.m_MSNetworkCfg.multiplier_network_type) {
        case LINEAR: 
	    this->msnet = new MSNetwork(2, "MSNetwork", stonne_cfg);
		std::cout << "MSNetwork" << std::endl;
	    break;
	case OS_MESH:
	    this->msnet = new OSMeshMN(2, "OSMesh", stonne_cfg);
		std::cout << "OSMeshMN" << std::endl;

	    break;
	default:
	    assert(false);
    }
    //switch(DistributionNetwork). It is possible to create instances of other DistributionNetworks.h
    this->dsnet = new DSNetworkTop(1, "DSNetworkTop", stonne_cfg);
    
    //Creating the ReduceNetwork according to the parameter specified by the user
    switch(stonne_cfg.m_ASNetworkCfg.reduce_network_type) {
    case ASNETWORK:
        this->asnet = new ASNetwork(3, "ASNetwork", stonne_cfg, outputASConnection); 
		std::cout << "ASNetwork" << std::endl;

        break;
    case FENETWORK:
        this->asnet = new FENetwork(3, "FENetwork", stonne_cfg, outputASConnection);
		std::cout << "FENetwork" << std::endl;

        break;
    case TEMPORALRN:
	this->asnet = new TemporalRN(3, "TemporalRN", stonne_cfg, outputASConnection);
	std::cout << "TemporalRN" << std::endl;

	break;
    default:
	assert(false);
    }

    this->collectionBus = new Bus(4, "CollectionBus", stonne_cfg); 
    this->lt = new LookupTable(5, "LookUpTable", stonne_cfg, outputASConnection, outputLTConnection);

    //switch(MemoryController). It is possible to create instances of other MemoryControllers
    switch(stonne_cfg.m_SDMemoryCfg.mem_controller_type) {
	case SIGMA_SPARSE_GEMM:
            this->mem = new SparseSDMemory(0, "SparseSDMemory", stonne_cfg, this->outputLTConnection);
			std::cout << "SparseSDMemory" << std::endl;

	    break;
	case MAERI_DENSE_WORKLOAD:
	    this->mem = new  SDMemory(0, "SDMemory", stonne_cfg, this->outputLTConnection);
		std::cout << "SDMemory" << std::endl;

	    break;
	case TPU_OS_DENSE:
	    this->mem = new  OSMeshSDMemory(0, "OSMeshSDMemory", stonne_cfg, this->outputLTConnection);
		std::cout << "OSMeshSDMemory" << std::endl;

	    break;
	default:
	    assert(false);
    }
    //Adding to the memory controller the asnet and msnet to reconfigure them if needed
    this->mem->setReduceNetwork(asnet);
    this->mem->setMultiplierNetwork(msnet); 

    //Calculating n_adders
    this->n_adders=this->ms_size-1; 
    //rsnet
	//read data
    this->connectMemoryandDSN();  //from dram to DNs
    this->connectMSNandDSN();	//from DNs to MNs
    this->connectMSNandASN();	//from MNs to RNs
	//write data
    this->connectASNandBus();	//from RNs to global buffer
    this->connectBusandMemory();//from global buffer to dram

    //STATISTICS
    this->n_cycles = 0;

}

Stonne::~Stonne() {
    delete this->dsnet;
    delete this->msnet;
    delete this->asnet;
    delete this->outputASConnection;
    delete this->outputLTConnection;
    delete this->lt;
    delete this->mem;
    delete this->collectionBus;
    if(layer_loaded) {
        delete this->dnn_layer;
    }
  
    if(tile_loaded) {
        delete this->current_tile;
    } 
}

//Connecting the DSNetworkTop input ports with the read ports of the memory. These connections have been created
//by the module DSNetworkTop, so we just have to connect them with the memory.
void Stonne::connectMemoryandDSN() {
    std::vector<Connection*> DSconnections = this->dsnet->getTopConnections();
    //Connecting with the memory
    this->mem->setReadConnections(DSconnections);
}

//Connecting the multipliers of the mSN to the last level switches of the DSN. In order to do this link correct, the number of 
//connections in the last level of the DSN (output connections of the last level switches) must match the number of multipliers. 
//The multipliers are then connected to those connections, setting a link between them. 
void Stonne::connectMSNandDSN() {
    std::map<int, Connection*> DNConnections = this->dsnet->getLastLevelConnections(); //Map with the DS connections
    this->msnet->setInputConnections(DNConnections);
     
}
//Connect the multiplier switches with the Adder switches. Note the number of ASs connection connectionss and MSs must be the identical

void Stonne::connectMSNandASN() {
    std::map<int, Connection*> RNConnections = this->asnet->getLastLevelConnections(); //Map with the AS connections
    this->msnet->setOutputConnections(RNConnections);

}

void Stonne::connectASNandBus() {
        std::vector<std::vector<Connection*>> connectionsBus = this->collectionBus->getInputConnections(); //Getting the CollectionBus Connections
        this->asnet->setMemoryConnections(connectionsBus); //Send the connections to the ReduceNetwork to be connected according to its algorithm
   
   
    
}

void Stonne::connectBusandMemory() {
    std::vector<Connection*> write_port_connections = this->collectionBus->getOutputConnections();
    this->mem->setWriteConnections(write_port_connections);
       
}

void Stonne::loadDNNLayer(Layer_t layer_type, std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G, unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, address_t input_address, address_t filter_address, address_t output_address, Dataflow dataflow) {
    assert((C % G)==0); //G must be multiple of C
    assert((K % G)==0); //G must be multiple of K
    assert(X>=R);
    assert(Y>=S);
    if((layer_type==FC) || (layer_type==GEMM)) {
        //assert((R==1) && (C==1) && (G==1) && (Y==S) && (X==1)); //Ensure the mapping is correct
    } 
    this->dnn_layer = new DNNLayer(layer_type, layer_name, R,S, C, K, G, N, X, Y, strides);

    this->layer_loaded = true;
    this->mem->setLayer(this->dnn_layer, input_address, filter_address, output_address, dataflow);
}

void Stonne::loadCONVLayer(std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G, unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, address_t input_address, address_t filter_address, address_t output_address) {
    loadDNNLayer(CONV, layer_name, R, S, C, K, G, N, X, Y, strides, input_address, filter_address, output_address, CNN_DATAFLOW);
    std::cout << "Loading a convolutional layer into STONNE" << std::endl;
}

void Stonne::loadFCLayer(std::string layer_name, unsigned int N, unsigned int S, unsigned int K, address_t input_address, address_t filter_address, address_t output_address)  {
     //loadDNNLayer(FC, layer_name, 1, S, 1, K, 1, N, 1, S, 1, input_address, filter_address, output_address, CNN_DATAFLOW);
    loadDNNLayer(FC, layer_name, 1, S, 1, K, 1, 1, N, S, 1, input_address, filter_address, output_address, CNN_DATAFLOW);
    std::cout << "Loading a FC layer into STONNE" << std::endl;
}

void Stonne::loadGEMM(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, metadata_address_t MK_metadata, metadata_address_t KN_metadata, address_t output_matrix, metadata_address_t output_metadata, Dataflow dataflow) {
    //Setting GEMM (from SIGMA) parameters onto CNN parameters:
    //N=N
    //S and X in CNN =K in SIGMA
    //K in CNN = M in SIGMA
    //input_matrix=KN 
    //filter_matrix = MK
    loadDNNLayer(GEMM, layer_name, 1, K, 1, M, 1, 1, N, K, 1, MK_matrix, KN_matrix, output_matrix, dataflow);
    std::cout << "Loading a GEMM into STONNE" << std::endl;
    this->mem->setSparseMetadata(MK_metadata, KN_metadata, output_metadata); 
    std::cout << "Loading metadata" << std::endl;
}

void Stonne::loadDenseGEMM(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, address_t output_matrix, Dataflow dataflow) {
    //Setting GEMM (from SIGMA) parameters onto CNN parameters:
    //N=N
    //S and X in CNN =K in SIGMA
    //K in CNN = M in SIGMA
    //input_matrix=KN
    //filter_matrix = MK
    loadDNNLayer(GEMM, layer_name, 1, K, 1, N, 1, 1, M, K, 1, MK_matrix, KN_matrix, output_matrix, dataflow);
    std::cout << "Loading a GEMM into STONNE" << std::endl;
}

void Stonne::loadSparseDense(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, address_t output_matrix, unsigned int T_N, unsigned int T_K) {
    //Setting GEMM (from SIGMA) parameters onto CNN parameters:
    //K in CNN=N here
    //C in CNN =K here
    //N in CNN = M here
    //input_matrix=MK 
    //filter_matrix = KN
    loadDNNLayer(SPARSE_DENSE, layer_name, 1, 1, K, N, 1, M, 1, 1, 1, MK_matrix, KN_matrix, output_matrix, SPARSE_DENSE_DATAFLOW);
    std::cout << "Loading a Sparse multiplied by dense GEMM into STONNE" << std::endl;
    /////To define in the new class
    this->mem->setSparseMatrixMetadata(MK_metadata_id, MK_metadata_pointer);
    std::cout << "Loading metadata" << std::endl;

    /////To define in the new class
    this->mem->setDenseSpatialData(T_N, T_K);
    std::cout << "Loading tile data" << std::endl;
}


void Stonne::loadGEMMTile(unsigned int T_N, unsigned int T_K, unsigned int T_M)  {
    //loadTile(1, T_K, 1, T_M, 1, T_N, 1, 1);
    loadTile(1, T_K, 1, T_N, 1, 1, T_M, 1);
    assert(this->layer_loaded && (this->dnn_layer->get_layer_type() == GEMM));   //Force to have the right layer with the GEMM parameters)
    std::cout << "Loading a GEMM tile" << std::endl;
}



//To dense CNNs and GEMMs 
void Stonne::loadTile(unsigned int T_R, unsigned int T_S, unsigned int T_C, unsigned int T_K, unsigned int T_G, unsigned int T_N, unsigned int T_X_, unsigned int T_Y_) {
    assert(this->layer_loaded);
    if(stonne_cfg.m_MSNetworkCfg.multiplier_network_type==LINEAR) {
        assert(this->ms_size >= (T_R*T_S*T_C*T_K*T_G*T_N*T_X_*T_Y_)); //There are enough mswitches
    }
    else {
        assert((this->stonne_cfg.m_MSNetworkCfg.ms_rows*this->stonne_cfg.m_MSNetworkCfg.ms_cols) >= (T_R*T_S*T_C*T_K*T_G*T_N*T_X_*T_Y_));
    }
    //Checking if the dimensions fit the DNN layer. i.e., the tile is able to calculate the whole layer.
    std::cout << "Loading Tile: <T_R=" << T_R << ", T_S=" << T_S << ", T_C=" << T_C << ", T_K=" << T_K << ", T_G=" << T_G << ", T_N=" << T_N << ", T_X'=" << T_X_ << ", T_Y'=" << T_Y_ << ">" << std::endl; 
 
    //Remove these lines if we want the architeture to compute the layer even if the tile does not fit. 
    // This will mean that some row, columns or output channels would remain without calculating. 
    if(stonne_cfg.m_SDMemoryCfg.mem_controller_type==MAERI_DENSE_WORKLOAD) { //Just for this maeri controller
        assert((this->dnn_layer->get_R() % T_R) == 0);    // T_R must be multiple of R
        assert((this->dnn_layer->get_S() % T_S) == 0);    // T_S must be multiple of S
        assert((this->dnn_layer->get_C() % T_C) == 0);    // T_C must be multiple of C
        assert((this->dnn_layer->get_K() % T_K) == 0);    // T_K must be multiple of K
        assert((this->dnn_layer->get_G() % T_G) == 0);    // T_G must be multiple of G
        assert((this->dnn_layer->get_N() % T_N) == 0);    // T_N must be multiple of N
        assert((this->dnn_layer->get_X_() % T_X_) == 0);  // T_X_ must be multiple of X_
        assert((this->dnn_layer->get_Y_() % T_Y_) == 0);  // T_Y_ must be multiple of Y_ 
    }

    //End check
    unsigned int n_folding = (this->dnn_layer->get_R() / T_R)*(this->dnn_layer->get_S() / T_S) * (this->dnn_layer->get_C() / T_C) ;
    bool folding_enabled = false; //Condition to use extra multiplier. Note that if folding is enabled but some type of accumulation buffer is needed this is false as no fw ms is needed. 
    if((n_folding > 1) && (this->stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled==0) && (this->stonne_cfg.m_ASNetworkCfg.reduce_network_type != FENETWORK)) { //If there is folding and the RN is not able to acumulate itself, we have to use an extra MS to accumulate
        folding_enabled = true; 
        //When there is folding we leave one MS free per VN aiming at suming the psums. In next line we check if there are
        // enough mswitches in the array to support the folding. 
        assert(this->ms_size >= ((T_R*T_S*T_C*T_K*T_G*T_N*T_X_*T_Y_) + (T_K*T_G*T_N*T_X_*T_Y_))); //We sum one mswitch per VN
    }
    this->current_tile = new Tile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_, folding_enabled);
    
    //Generating the signals for the reduceNetwork and configuring it. The asnet->configureSignals will call its corresponding compiler to generate the signals and allocate all of them
    if(this->stonne_cfg.m_MSNetworkCfg.multiplier_network_type != OS_MESH) { //IN TPU the configuration is done in the mem controller
        this->asnet->configureSignals(this->current_tile, this->dnn_layer, this->ms_size, n_folding); //Calling the ART to configure the signals with them previously generated 
    //Getting MN signals
        this->msnet->configureSignals(this->current_tile, this->dnn_layer, this->ms_size, n_folding);
    }
    //Setting the signals to the corresponding networks

    //If stride > 1 then all the signals of ms_fwreceive_enabled and ms_fwsend_enabled must be disabled since no reuse between MSwitches can be done. In order to not to incorporate stride
    //as a tile parameter, we leave the class Tile not aware of the stride. Then, if stride exists, here the possible enabled signals (since tile does not know about tile) are disabled.
    this->tile_loaded = true;
    this->mem->setTile(this->current_tile);
}

void Stonne::loadFCTile(unsigned int T_S, unsigned int T_N, unsigned int T_K)  {
    //loadTile(1, T_S, 1, T_K, 1, T_N, 1, 1);
    loadTile(1, T_S, 1, T_K, 1, 1, T_N, 1);
    assert(this->layer_loaded && (this->dnn_layer->get_layer_type() == FC));   //Force to have the right layer with the FC parameters)
    std::cout << "Loading a FC tile" << std::endl;
}

//modified by wangjs
void Stonne::run(string layer_name) {
    //Execute the cycles

    this->cycle(layer_name);
}
//modified end

//added by wangjs
string index_layer_num(int num)
{
	stringstream ss;
	ss << setw(4) << setfill('0') << num;
	string str;
	ss >> str;         //将字符流传给 str
	//str = ss.str();  //也可以
	return str;
}
//added end



//modified by wangjs
void Stonne::cycle(string layer_name) {
	 

	config_file = this->config_file_;
	std::string file_name = config_file.substr(config_file.find_last_of('/') + 1);  // 获取文件名部分
	std::string x_y = file_name.substr(0, file_name.find('.'));  // 去除文件扩展名

	output_dir = output_dir_ + this->arch_name + "_" + this->model_name + "_" + x_y;
	std::string command = "mkdir -p " + output_dir;
	system(command.c_str());

	CPU* cpu = new TraceBasedCPU(config_file, output_dir);
	cpu->arch_name = this->arch_name;
	cpu->model_name = this->model_name;
	cpu->dram_name = x_y;
	//cpu->test_layer_num++;

    while(!execution_finished || !cpu->req_fifo.empty() || !cpu->l2_fifo->isEmpty() || !cpu->dram_fifo->isEmpty())
	{
		if (!execution_finished) {
			this->mem->cycle(cpu);
			this->collectionBus->cycle(cpu);
			this->asnet->cycle();
			this->lt->cycle();
			this->msnet->cycle();
			this->dsnet->cycle();
			execution_finished = this->mem->isExecutionFinished();
		}
		 

		this->mem_partition_cycle(cpu); //L2 MEE DRAM 

		cpu_finished = cpu->isFinished();
		cpu->n_cycles++; 
		
    } 

	string index_layer_ = index_layer_num(index_layer++); 
	printSystemStats(cpu, layer_name, index_layer_);

 	delete cpu;
}
//modified end

//added by wangjs
void Stonne::mem_partition_cycle(CPU * cpu)
{ 

	if (!cpu_finished || !cpu->req_fifo.empty()) { 
		cpu->setAccessConfig();
	}

	cpu->l2->cache_clock(cpu);
	cpu->mac_cache->cache_clock(cpu);
	cpu->ctr_cache->cache_clock(cpu);
	cpu->bmt_cache->cache_clock(cpu);
	cpu->AccessDram(); 

		
	//cpu->partition_cycle++; 
}
//added end

//void Stonne::ResetSystemStats() {
//	mem_load_num = 0;	    //与DRAM交互的load指令数
//	mem_store_num = 0;		//与DRAM交互的store指令数
// }

//modified by wangjs
void Stonne::printSystemStats(CPU *cpu, string layer_name, string index_layer_) {
	//cpu->PrintStats();

	cpu->GetDelay();
	cpu->l2->PrintCache(layer_name, index_layer_, output_dir);
	cpu->ctr_cache->PrintCache(layer_name, index_layer_, output_dir);
	cpu->bmt_cache->PrintCache(layer_name, index_layer_, output_dir);
	cpu->mac_cache->PrintCache(layer_name, index_layer_, output_dir);

	global_total_read_delay += cpu->total_read_delay;
 	global_total_read_num += cpu->total_read_num; 
	global_avg_read_delay = static_cast<double>(global_total_read_delay) / global_total_read_num;
	global_n_cycles += cpu->n_cycles;
	 


	string cpu_out_file = output_dir + "/delay.csv";
	ofstream ofs;
	ofs.open(cpu_out_file, ios::out | ios::app);
	if (index_layer_ == "0001") {
		ofs << "total_delay" << "," << "avg_delay" << "," << "total_num" << "," << "global_cycle" << "," << endl;
	}
 	ofs << global_total_read_delay << "," << global_avg_read_delay << "," << global_total_read_num << "," << global_n_cycles << "," << endl;
	ofs.close();
}
//modified end

//General function to print all the STATS
void Stonne::printStats() {
    std::cout << "Printing stats" << std::endl;

    std::ofstream out; 
    unsigned int num_ms = this->stonne_cfg.m_MSNetworkCfg.ms_size;
    unsigned int dn_bw = this->stonne_cfg.m_SDMemoryCfg.n_read_ports;
    unsigned int rn_bw = this->stonne_cfg.m_SDMemoryCfg.n_write_ports;
    const char* output_directory=std::getenv("OUTPUT_DIR");
    std::string output_directory_str="";
    if(output_directory!=NULL) {
        std::string env_output_dir(output_directory);
        output_directory_str+=env_output_dir+"/";
    }

    out.open(output_directory_str+"output_stats_layer_"+this->dnn_layer->get_name()+"_architecture_MSes_"+std::to_string(num_ms)+"_dnbw_"+std::to_string(dn_bw)+"_"+"rn_bw_"+std::to_string(rn_bw)+"timestamp_"+std::to_string((int)time(NULL))+".txt"); //TODO Modify name somehow
    unsigned int indent=IND_SIZE;
    out << "{" << std::endl;

        //Printing input parameters
        this->stonne_cfg.printConfiguration(out, indent);
        out << "," << std::endl;

        //Printing layer configuration parameters
        this->dnn_layer->printConfiguration(out, indent);
        out << "," << std::endl;

        //Printing tile configuration parameters
       // this->current_tile->printConfiguration(out, indent);
        //out << "," << std::endl;   
        
        //Printing ASNetwork configuration parameters (i.e., ASwitches configuration for these VNs, flags, etc)
        this->asnet->printConfiguration(out, indent);
        out << "," << std::endl;
  
        this->msnet->printConfiguration(out, indent);
        out << "," << std::endl;

        
        //Printing global statistics
        this->printGlobalStats(out, indent);
        out << "," << std::endl;        

        //Printing all the components
        this->dsnet->printStats(out, indent);  //DSNetworkTop //DSNetworks //DSwitches
        out << "," << std::endl;
        this->msnet->printStats(out, indent);
        out << "," << std::endl;
        this->asnet->printStats(out, indent);
        out << "," << std::endl;
        this->mem->printStats(out, indent);
        out << "," << std::endl;
        this->collectionBus->printStats(out, indent);
        out << std::endl;
        
     
    
    out << "}" << std::endl;
    out.close();
	std::cout << "Printing stats" << std::endl;
}

void Stonne::printEnergy() {
    std::ofstream out;

    unsigned int num_ms = this->stonne_cfg.m_MSNetworkCfg.ms_size;
    unsigned int dn_bw = this->stonne_cfg.m_SDMemoryCfg.n_read_ports;
    unsigned int rn_bw = this->stonne_cfg.m_SDMemoryCfg.n_write_ports;

    const char* output_directory=std::getenv("OUTPUT_DIR");
    std::string output_directory_str="";
    if(output_directory!=NULL) {
        std::string env_output_dir(output_directory);
        output_directory_str+=env_output_dir+"/";
    }

    out.open(output_directory_str+"output_stats_layer_"+this->dnn_layer->get_name()+"_architecture_MSes_"+std::to_string(num_ms)+"_dnbw_"+std::to_string(dn_bw)+"_"+"rn_bw_"+std::to_string(rn_bw)+"timestamp_"+std::to_string((int)time(NULL))+".counters"); //TODO Modify name somehow
    unsigned int indent=0;
    out << "CYCLES=" <<  this->n_cycles << std::endl; //This is to calculate the static energy
    out << "[DSNetwork]" << std::endl;
    this->dsnet->printEnergy(out, indent);  //DSNetworkTop //DSNetworks //DSwitches

    out << "[MSNetwork]" << std::endl;
    this->msnet->printEnergy(out, indent);

    out << "[ReduceNetwork]" << std::endl;
    this->asnet->printEnergy(out, indent);

    out << "[GlobalBuffer]" << std::endl;
    this->mem->printEnergy(out, indent);

    out << "[CollectionBus]" << std::endl;
    this->collectionBus->printEnergy(out, indent);

    out << std::endl;
    out.close();

}

//Local function to the accelerator to print the globalStats
void Stonne::printGlobalStats(std::ofstream& out, unsigned int indent) {
    //unsigned int n_mswitches_used=this->current_tile->get_VN_Size()*this->current_tile->get_Num_VNs();
    //float percentage_mswitches_used = (float)n_mswitches_used / (float)this->stonne_cfg.m_MSNetworkCfg.ms_size;
    out << ind(indent) << "\"GlobalStats\" : {" << std::endl; //TODO put ID
    //out << ind(indent+IND_SIZE) << "\"N_mswitches_used\" : " << n_mswitches_used << "," << std::endl;
    //out << ind(indent+IND_SIZE) << "\"Percentage_mswitches_used\" : " << percentage_mswitches_used << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"N_cycles\" : " << this->n_cycles << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability

}


