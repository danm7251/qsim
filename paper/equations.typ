#let hadamard_matrix = $
    frac(1, sqrt(2))
    mat(delim: "[", 1, 1; 1, -1;)
$

#let identity_matrix = $
    mat(delim: "[", 1, 0; 0, 1;)       
$

#let x_matrix = $
    mat(delim: "[", 0, 1; 1, 0;)                  
$

#let y_matrix = $
    mat(delim: "[", 0, -i; i, 0;)                 
$

#let z_matrix = $
    mat(delim: "[", 1, 0; 0 -1;)
$

#hadamard_matrix
#identity_matrix
#x_matrix
#y_matrix
#z_matrix