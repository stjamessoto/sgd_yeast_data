Y1000+ Dataset
https://y1000plus.wei.wisc.edu/
Corresponding Authors: Chris Todd Hittinger (cthittinger@wisc.edu) and Antonis Rokas (antonis.rokas@vanderbilt.edu) 

Analysis:

For each of the 1,154 yeast genome assemblies (including previously sequenced 
genomes), repetitive sequences were identified and softmasked using RepeatMasker 
v4.1.2  with the “-species” option set to “saccharomycotina”. Protein-coding 
genes were annotated using BRAKER v2.1.6 ​(Brůna et al., 2021)​, with all 
Saccharomycetes protein sequences in the OrthoDB v10 as homology evidence, and 
AUGUSTUS v3.4.0 ​(Stanke et al., 2008)​ as well as GeneMark-EP+ v4.6.1 ​(Brůna et 
al., 2020)​ as ab initio gene predictors. The BRAKER pipeline was run in the EP 
mode to process all protein homology-evidence using ProtHint v2.6.0 ​(Brůna et 
al., 2020)​, and the “--fungus” option was turned on to run GeneMark-EP+ with the
branch point model for fungal genomes. For genes with multiple transcripts, only
the longest transcript was retained.  The peptide translations were all 
conducted using the standard nuclear codon table (Genetic Code Translation 
Table 1).  


Folder Structure:
y1000p_cds_files.tar.gz - Fasta files of the nucleotide sequences for the gene annotations
	assembly_fullID.final.cds
	
y1000p_gff3_files.tar.gz - GFF3 files of the gene annotations
	assembly_fullID.final.gff3
	
y1000p_gtf_files.tar.gz - GTF files of the gene annotations
	assembly_fullID.final.gtf	

y1000p_pep_files.tar.gz - Fasta files of the amino acid sequences for the gene annotations
	assembly_fullID.final.pep

