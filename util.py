#choices = [-0.1,0,0.1]
def permutate(all_lists, the_list, choices, n):
    if (len(the_list) == n):
        all_lists.append(the_list.copy()) 
        return    all_lists        
    for c in choices:
        the_list.append(c)
        permutate(all_lists, the_list, choices,  n)
        the_list.pop()
    return all_lists
#print(permutate([],[],choices, 5))