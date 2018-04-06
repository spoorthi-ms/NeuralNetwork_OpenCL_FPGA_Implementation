**********************************************************************************************************
DESCRIPTION: Kernel code for Pool Layer 2 of Cifarnet.
             Consists of: 3 Convolution Layers, 2 Fully Connected Layers, 3 Pool Layers.

			This code is developed in OpenCL using Vivado HLS. 
			Bitstream is generated using Vivado.
			The host code is developed in Xilinx SDK.


Author: Spoorthi Mysore Shivakumar (shivakumar.spoorthi@gmail.com)

**********************************************************************************************************
__kernel void __attribute__ ((reqd_work_group_size(1024,1,1))) p2(__global float *Layer2_Neurons_GPU, __global float *Layer2_pool_GPU, __global float *Layer2_Features)
		{
	int z = get_global_id(0);
	int row = z/32;
	int col = z%32;
	float avg = 0;
	int count = 0;
	{
	        for(int output =0;output < 32 ;output++)
	        {
	            if((row%2 != 0) && (row<16))
	            {
	                if((col%2 != 0) && (col<16))
	                {
	                    for(int i = row-1; i <= row+1; i++)
	                    {
				if(i>15) break;
	                        for(int j = col-1; j <= col+1; j++)
	                        {
				    if(j>15) break;
	                            avg+= Layer2_Neurons_GPU[output*16*16 + i*16 + j];
				    count = count + 1;

	                        }
	                    }
	                    Layer2_pool_GPU[output*8*8+(row-1)*4+(col-1)/2] = avg/count;
	                    avg = 0;
			    count=0;
	                }
	            }
	        }
	    }

}
