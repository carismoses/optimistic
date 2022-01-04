# Use base stacking image
FROM carismoses/optimistic:latest

# run training
COPY random_actions.sh .
RUN chmod +x random_actions.sh
CMD ./random_actions.sh
