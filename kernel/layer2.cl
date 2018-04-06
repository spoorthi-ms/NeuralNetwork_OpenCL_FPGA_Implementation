/**********************************************************************************************************
DESCRIPTION: Kernel code for Layer 2 (Convolution Layer) of Cifarnet.
             Consists of: 3 Convolution Layers, 2 Fully Connected Layers, 3 Pool Layers.

			This code is developed in OpenCL using Vivado HLS. 
			Bitstream is generated using Vivado.
			The host code is developed in Xilinx SDK.


Author: Spoorthi Mysore Shivakumar (shivakumar.spoorthi@gmail.com)

**********************************************************************************************************/


__kernel void __attribute__ ((reqd_work_group_size(256,1,1))) sL(__global float *Layer2_Weights_CPU, __global float *Layer2_pool_GPU, __global float *Layer2_Features)
		{
	int z = get_global_id(0);
	int x = z/16;
	int y = z%16;

	float Features = 0;
	int feat=0;
	for(feat=0; feat<32; feat++)
	{
		Features = 0;
		for(int n=0; n<32; n++)
		{
			if(x<16)
			{
				if(y<16)
				{
					float result = 0;
					for(int i = x-2; i<=x+2; i++)
					{
						for(int j=y-2; j<=y+2; j++)
						{
							int x_index = i-x+2;
							int y_index = j-y+2;
							int m = (y_index)+(x_index)*5;
							if(i<0 || j<0)
							{
								result+=0;
							}
							else if(j>15 || i>15)
							{
								result+=0;
							}
							else
							{
								result+= Layer2_pool_GPU[n*16*16 + (x_index+x-2)*16 + (y_index+y-2)]*Layer2_Weights_CPU[m+feat*25*32+n*25];
							}
						}
					}
					Features += result;
				}
			}
		}
		//ReLU activation function computation
		if(Features<0)
		{
			Features = 0;
		}

		Layer2_Features[feat*16*16 + x*16 + y] = Features;
	}

}
