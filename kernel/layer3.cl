/**********************************************************************************************************
DESCRIPTION: Kernel code for Layer 3 (Convolution Layer) of Cifarnet.
             Consists of: 3 Convolution Layers, 2 Fully Connected Layers, 3 Pool Layers.

			This code is developed in OpenCL using Vivado HLS. 
			Bitstream is generated using Vivado.
			The host code is developed in Xilinx SDK.


Author: Spoorthi Mysore Shivakumar (shivakumar.spoorthi@gmail.com)

**********************************************************************************************************/


__kernel void __attribute__ ((reqd_work_group_size(8,8,1))) tl(__global float *Layer3_Weights_CPU, __global float *Layer2_pool_GPU, __global float *Layer3_Features)
{

	int x = get_global_id(0);
	int y = get_global_id(1);

	float Features = 0;

	for(int f=0; f<64; f++)
	{
		Features = 0;
		for(int n=0; n<32; n++)
		{
			if(x<8)
			{
				if(y<8)
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
							else if(j>7 || i>7)
							{
								result+=0;
							}
							else
							{
								result+= Layer2_pool_GPU[n*8*8 + (x_index+x-2)*8 + (y_index+y-2)]*Layer3_Weights_CPU[m+f*25*32+n*25];
							}
						}
					}
					Features += result;
				}
			}
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		Layer3_Features[f*8*8 + x*8 + y] = Features;
	}


}
