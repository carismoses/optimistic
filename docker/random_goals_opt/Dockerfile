# Use base stacking image
FROM carismoses/optimistic:latest

# run training
COPY random_goals.sh .
RUN chmod +x random_goals.sh
CMD ./random_goals.sh
