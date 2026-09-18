from fractions import Fraction
u=Fraction(1,1<<24);q=1+Fraction(1,1<<16);rho=Fraction(84,1<<25)
t=q*(1+rho)/(1-u)**8
for n in range(1,129):assert t**n<1+Fraction(n,1<<15)
print({'checked_lengths':128,'per_step_upper_factor':float(t),'maximum_total_factor':float(t**128),'final_allowance_factor':1+128*2**-15,'status':'pass'})
