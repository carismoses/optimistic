# Use base stacking image
FROM carismoses/optimistic:latest

# copy experiment log into container
WORKDIR /
COPY sequential_goals-20220105-143004 .

# run training
COPY restart_sequential_goals.sh .
RUN chmod +x restart_sequential_goals.sh
CMD ./restart_sequential_goals.sh
