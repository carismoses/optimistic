# Use base stacking image
FROM carismoses/optimistic:latest

# run training
COPY engineered_goals_dist.sh .
RUN chmod +x engineered_goals_dist.sh
CMD ./engineered_goals_dist.sh
