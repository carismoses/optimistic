# Use base stacking image
FROM carismoses/optimistic:latest

# run training
COPY sequential_plans.sh .
RUN chmod +x sequential_plans.sh
CMD ./sequential_plans.sh
