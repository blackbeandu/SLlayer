#pragma once

// don't touch this file

#include <SoyPlugin.h>
#include <string>
#include <NvInfer.h>
class SAudio_sb : public SPlugin
{
public:
	SAudio_sb() = default;
	~SAudio_sb() = default;

	virtual std::string pluginType() final;
	virtual std::string pluginVersion() final;
	virtual std::string pluginDesc() final;
	virtual int isShareAct() final { return ACT_IMPL::SHARE; }
	virtual int initParam(SLayer& layer, std::string& log_str) final;
	virtual void freeParam(SLayer& layer) final;
	//virtual int calc_weight_shape(SLayer& flayer, sv_map& player, std::string& log) final;
	virtual int load_weight(std::vector<STensor*>& weights, SLayer& layer, std::ifstream& wfs, std::string& log_str) final;
	virtual int calc_output_shape(std::vector<std::vector<int>>& oshapes, std::vector<std::vector<int>>& ishapes, SLayer& layer, std::string& log_str) final;
	virtual void get_model_arch(std::string& log, SLayer& layer) final;
	virtual int build_layer(std::vector<void*>& iplugins, std::vector<void*>& outputITensors, std::vector<void*>& inputITensors, SLayer& layer, SSect& model, void* network, std::string& log_str) final;
	virtual void destroy_iplugins(std::vector<void*>& iplugins) final
	{
		for (int pidx = 0; pidx < iplugins.size(); pidx++) {
			nvinfer1::IPluginV2DynamicExt* iplugin = (nvinfer1::IPluginV2DynamicExt*)iplugins[pidx];
			iplugin->destroy();
		}
	}
	virtual void destroy_resources(std::vector<STensor*>& weights, std::vector<void*>& resources) final
	{
		destroy_weight_tensor(weights);
		for (int ridx = 0; ridx < resources.size(); ridx++) {
			free(resources[ridx]);
		}
	}
};
