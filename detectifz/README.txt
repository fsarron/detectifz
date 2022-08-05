### class Data should represent the data used by detectifz. 
__init__ : reads the data (galaxy catalogue, PDFs) 


### class DETECTIFz define the functions to run the full group detection on a given Data object 
with a given set of parameters (zmin, zmax, pixdeg, etc.)

self.__init__(): should check that the Data object is valid for group detection
i.e. Data needs to contain 

self.run():  should run all the steps of the detection in order 
(similarly to run_DETECTIFz in DETECTIFz2d_data.py)

To run full DETETECTIFz + members right now I need : 
        det : sigz68_z 
        membership : sigz68_Mz (convolve pdzclus for likelihood), sigz95_Mz (smooth N(M,z) for prior)). 
        
        Note: for prior, I could replace  - smooth N(M,z) with Tophat(sigz95_Mz)  
                                      by  - N(M,z) convolved with Normal(sigz68_Mz)
                                      
        I think it makes more sense from a "probability theory" stand point -- and this speed-up get_sig()