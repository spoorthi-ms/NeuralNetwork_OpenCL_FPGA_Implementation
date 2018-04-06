/**********************************************************************************************************
DESCRIPTION: Kernel code for Layer 5 (Fully Connected Layer) of Cifarnet.
             Consists of: 3 Convolution Layers, 2 Fully Connected Layers, 3 Pool Layers.

			This code is developed in OpenCL using Vivado HLS. 
			Bitstream is generated using Vivado.
			The host code is developed in Xilinx SDK.


Author: Spoorthi Mysore Shivakumar (shivakumar.spoorthi@gmail.com)

**********************************************************************************************************/


__kernel void __attribute__ ((reqd_work_group_size(32,1,1))) fifL(__global float *Layer5_Weights_CPU, __global float *Layer5_Features, __global float *Layer4_Features)

{
	int n = get_global_id(0);

	if(n<9)
		{
			float result = 0;
			for(int f=0; f<64; f++)
			{
				result+= Layer4_Features[f] * Layer5_Weights_CPU[f+n*64];
			}
			Layer5_Features[n] = result;
			result = 0;
		}

}
