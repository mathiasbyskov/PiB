

class GSM_data:

	import os
	import pandas as pd


	def __init__(self, samples, download = True, filepath = './', silent = False, print_file = False):
		self.samples = samples
		self.destdir = filepath
		self.filepath = filepath
		self.silent = silent
		self.print_file = print_file
		
		if download:
			self.download_GSM_sample(samples, filepath, silent = silent, print_file = print_file)

		self.df = self.samples_to_dataframe(samples, filepath = self.filepath)
			

	def download_GSM_sample(self, samples, destdir = './', silent = False, print_file = False):
	    """ Method for downloading the samples into a .txt file to the specified filepath. """
	    
	    import GEOparse
	    
	    if print_file:
	    	f = open("downloaded-samples.txt", "w")
	    
	    for sample in enumerate(samples):
	        GEOparse.get_GEO(sample[1], destdir=destdir, silent=silent)
	        
	        if print_file: 
	        	f.write("{}\n".format(sample[1]))
	        
	        print("Sample: {} downloaded ({} out of {})".format(sample[1], sample[0]+1, len(samples)))
	    
	    if print_file:
	    	f.close()
	    return "All samples downloaded to directory!"


	def samples_to_dataframe(self, samples, filepath = './', silent = False):
	    """ Method to read existing samples and create clean dataframe. """
	    
	    import pandas as pd
	    import GEOparse
	    
	    for sample in enumerate(samples):
	        if sample[0] == 0:
	            df = GEOparse.get_GEO(filepath = str(filepath + sample[1] + '.txt'), silent=silent).table
	            df = df[['ID_REF', 'VALUE']].sort_values('ID_REF')
	            df = df.pivot_table(columns = 'ID_REF')
	            df['Sample'] = sample[1]
	        else:
	            row = GEOparse.get_GEO(filepath = str(filepath + sample[1] + '.txt'), silent=silent).table
	            row = row[['ID_REF', 'VALUE']].sort_values('ID_REF')
	            row = row.pivot_table(columns = 'ID_REF')
	            row['Sample'] = sample[1]
	            df = pd.concat([df, row], join = 'outer')
	        
	        print("Sample: {} added ({} out of {})".format(sample[1], sample[0]+1, len(samples)))
	    return df
