# DDBP

# Branches and experiments
1. **comparison**  
This branch contains the code for an experiment with a simple autoencoder.  
2. **comparison_fixed**  
This branch is a further modification of **comparision**, during the fine tuning phase the weights in the already pretrained autoencoder are frozen and cannot be longer modified.  
3. **altered_comparison**  
This experiment is based on the approach described in [1], the network was modified accordingly to use the advantages of the dedicated connections described in the paper.  
4. **altered_comparison_fixed**  
This branch contains a similar modification as branch **comparison_fixed** but based on the experiment **altered_comparison**, the pretrained weights in the encoding layer are no longer modified during the last phase.



# References
[1]. Learning without human expertise: a case study of the double dummy bridge problem, Mossakowski K1, Mańdziuk J.
