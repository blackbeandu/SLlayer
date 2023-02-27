// https://www.chriswirz.com/software/cpp-plugin-architecture-in-ubuntu

#include <SoyPlugin.h>
#include <SoyUtil.h>
#include <SoyCUDA.h>
#include <string>
#include <chrono>
#include <limits>
#include <algorithm>
#include <NvInfer.h>
#include <SoyAct.h>
#include <cinttypes>
#include <cassert>
using namespace std;
using namespace chrono;

#include "cuda_runtime.h"
#include "audio_sb.hpp"
#include "plugin_audio_sb.h"

struct SParam {
	SParam(){}
	// here, modify and define your custom attributes
	std::string mode;
};

// DO NOT touch below 5 functions
SPlugin* plugin_entry(){ return (SPlugin*)new SAudio_sb; }
string SAudio_sb::pluginType() { return "audio_sb"; }
string SAudio_sb::pluginVersion() { return "2022.05.01.001"; }
string SAudio_sb::pluginDesc() { return  "audio_sb for only fastspeech"; }
void SAudio_sb::freeParam(SLayer& layer) { delete (SParam*)layer.param; }

int SAudio_sb::initParam(SLayer& layer, string& log_str)
{
	SParam* p = new(nothrow) SParam;
	if (p == nullptr) {
		log_str = format("[E] (L:%d,C:%d) param memory allocation error, in initParam of %s\n", layer.sect.line, layer.sect.col, pluginType());
		return -1;
	}
	layer.param = (void*)p;
	SParam& param = *p;

	// here, your code!,
	// be careful that you can refer NOT "layer.inputs[x]->shape" BUT "layer.inputs[x]->opt_shape"
    param.mode = sect2s(layer.sect, "mode");
	return 0;
}
int SAudio_sb::calc_output_shape(std::vector<std::vector<int>>& oshapes, std::vector<std::vector<int>>& ishapes, SLayer& layer, string& log_str)
{
    // opt_shape and max_shape are calculated in this layer.
	SParam& param = *(SParam*)layer.param;

	std::vector<int> oshape = ishapes[0];

	// your code
	oshapes.push_back(oshape);

	return 0;
}

int SAudio_sb::load_weight(std::vector<STensor*>& weights, SLayer& layer, std::ifstream& wfs, std::string& log_str)
{
	SParam& param = *(SParam*)layer.param;

	return 0;
}

void SAudio_sb::get_model_arch(string& arch, SLayer& layer)
{
	SParam& param = *(SParam*)layer.param;

    arch = format("mode=%s", param.mode);
}

int SAudio_sb::build_layer(std::vector<void*>& iplugins, std::vector<void*>& outputITensors, std::vector<void*>& inputITensors, SLayer& layer, SSect& model, void* network, std::string& log_str)
{
	SParam& param = *(SParam*)layer.param;
	nvinfer1::INetworkDefinition& net = *(nvinfer1::INetworkDefinition*)network;
	nvinfer1::ITensor* input = (nvinfer1::ITensor*)inputITensors[0];
    nvinfer1::ITensor* output = nullptr;



	nvinfer1::IPluginCreator* pluginCreator = getPluginRegistry()->getPluginCreator(AUDIO_SB_PLUGIN_NAME, AUDIO_SB_PLUGIN_VERSION);
	char* audio_sb_plugin_name = AUDIO_SB_PLUGIN_NAME;

	int mode = get_audio_sb_mode(param.mode);
	PAudioSb fc{ mode };
	nvinfer1::IPluginV2DynamicExt* audio_sb_plugin = (nvinfer1::IPluginV2DynamicExt*)pluginCreator->createPlugin(audio_sb_plugin_name, (nvinfer1::PluginFieldCollection*)&fc);
	iplugins.emplace_back(audio_sb_plugin);

	nvinfer1::ITensor* audio_0 = (nvinfer1::ITensor*)inputITensors[1];
	nvinfer1::ITensor* audio_1 = (nvinfer1::ITensor*)inputITensors[2];
	std::vector<nvinfer1::ITensor*> data{ input, audio_0, audio_1 };
	nvinfer1::IPluginV2Layer* audio_sb_layer = net.addPluginV2(data.data(), (int)data.size(), *audio_sb_plugin);
	output = audio_sb_layer->getOutput(0);


	assert(&output->getDimensions() != nullptr); // for debug
	outputITensors.push_back(output);
	return 0;
}

/*
fastspeech 에서만 사용
model.py 265라인의 다음 코드를 cuda로 구현
	s = output[:, n_half:, :]
	b = output[:, :n_half, :]
	audio_1 = (audio_1 - b) / torch.exp(s)
	audio = torch.cat([audio_0, audio_1], 1)

입력은 3개(이전layer출력, AUDIO_0, AUDIO_1)이고, 출력크기는 이전 layer의 크기와 동일
*/