# Use base stacking image
FROM carismoses/optimistic:latest

# run training
COPY random_goals_opt.sh .
RUN chmod +x random_goals_opt.sh
CMD ./random_goals_opt.sh
