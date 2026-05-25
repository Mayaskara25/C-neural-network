MEMORY OWNERSHIP :
anything to do with the layer is cleared by the layer ;
any other stuff is cleared by the caller;
keep an eye on what is being returned - most of the returned stuff here have pointers in them -i.e you have to look at two seperate things:
they are 
1)when the caller calls a fxn that retrns a struct or anything that has something malloced ,  what happens to the memory that was previously linked to it?
  i.e create_dense_layer() makes and allocs memory for every member matrix in the layer , but the forward pass makes and puts new stuff in input , output , and z - what happens to the memeory that these 
structs were linked to earlier ?- REMEBER TO FREE tHEM - i.e free_mat(layer->input); layer->input = copy_mat(input);

2)when a function returns a struct that is a member of the layer , the CALLER DOES NOT OWN THE MEMORY , matrix x = forward_pass(...) ; this returns layer->output to x , now both x and layer->outpu point to
the same memory location , so DO NOT FREE MATRIX X , just freeing the dense_layer sould free the heap pointed by x aswell.
 
