/**********************************************************************************************************
DESCRIPTION: Kernel code for Layer 4 (Fully Connected Layer) of Cifarnet.
             Consists of: 3 Convolution Layers, 2 Fully Connected Layers, 3 Pool Layers.

			This code is developed in OpenCL using Vivado HLS. 
			Bitstream is generated using Vivado.
			The host code is developed in Xilinx SDK.


Author: Spoorthi Mysore Shivakumar (shivakumar.spoorthi@gmail.com)

**********************************************************************************************************/


__kernel void __attribute__ ((reqd_work_group_size(64,1,1))) foL(__global float *Layer4_Weights_CPU, __global float *Pool3_Layer_Features, __global float *Layer4_Features)

{
	int n = get_global_id(0);

	{
		float result = 0;
		for(int f=0; f<64; f++)
		{
			for(int x=0; x<4; x++)
			{
				for(int y=0; y<4; y++)
				{
					result+= Pool3_Layer_Features[f*4*4 +x*4 + y] * Layer4_Weights_CPU[y+(x*4)+(f*4*4)+(n*4*4*64)];
				}
			}
		}
		Layer4_Features[n] = result;
	}
}
