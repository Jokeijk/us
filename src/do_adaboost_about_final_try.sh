#/home/dli/canopy/bin/python adaboost_about_final_try.py 5 1    
#/home/dli/canopy/bin/python adaboost_about_final_try.py 5 100
#/home/dli/canopy/bin/python adaboost_about_final_try.py None 1
#/home/dli/canopy/bin/python adaboost_about_final_try.py None 100
#/home/dli/canopy/bin/python adaboost_about_final_try.py 6 1
#/home/dli/canopy/bin/python adaboost_about_final_try.py 6 100


#predict
seq 30 |parallel -j30 /home/dli/canopy/bin/python adaboost_about_final_try.py {}
