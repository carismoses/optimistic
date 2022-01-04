# Use base stacking image
FROM carismoses/optimistic:latest

# run training
COPY sequential_goals.sh .
RUN chmod +x sequential_goals.sh
CMD ./sequential_goals.sh
