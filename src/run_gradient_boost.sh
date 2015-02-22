seq 1000 10 1200 | parallel -j21 --tag /home/dli/canopy/bin/python gradient_boost.py {} | tee run_gradient_boost.log
