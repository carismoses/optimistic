# Use base stacking image
FROM carismoses/optimistic:latest

# add all datasets
COPY datasets .

COPY train_models.sh .
RUN chmod +x train_models.sh
CMD ./train_models.sh

