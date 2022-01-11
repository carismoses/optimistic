# Use base stacking image
FROM carismoses/optimistic:latest

# run training
COPY engineered_goals_size.sh .
RUN chmod +x engineered_goals_size.sh
CMD ./engineered_goals_size.sh
