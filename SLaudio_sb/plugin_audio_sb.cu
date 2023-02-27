#include "cuda_runtime.h"
#include <math.h>
#include <stdio.h>
#include <cinttypes>
#include <NvInfer.h>
#include "plugin_audio_sb.h"

#define BLOCK_SIZE 128
__global__ void kernel_audio_sb(float* output, float* input, float* audio_0, float* audio_1, int S, int E, int tcount)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= tcount) return;

	int e_idx = pos % E;
	int idx = pos / E;
	int s_idx = idx % S;
	int b_idx = idx / S;

	if (s_idx < S / 2) {
		int source_idx = b_idx*S/2*E + s_idx*E +e_idx;
		output[pos] = audio_0[source_idx];
	}
	else {
		int b_val_idx = b_idx * S/2 * E + (s_idx-S/2) * E + e_idx;
		float b_val = audio_1[b_val_idx] - input[b_val_idx];
		float s_val = expf(-input[pos]);
		output[pos] = b_val * s_val;
	}
}

void cuda_audio_sb(float* output, float* input, float* audio_0, float* audio_1, int N, int S, int E, cudaStream_t stream)
{
	int tcount = N * S * E;
	int block = BLOCK_SIZE; // 128
	int grid = (tcount + block - 1) / block;
	//int shm_bytes = (E + block / 2 + block / 2) * sizeof(float); // sum, sum2
	kernel_audio_sb << < grid, block, 0, stream >> > (output, input, audio_0, audio_1, S, E, tcount);
}