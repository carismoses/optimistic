# Use base stacking image
FROM carismoses/optimistic:latest

# add all models
COPY models .

COPY make_datasets.sh .
RUN chmod +x make_datasets.sh
CMD ./make_datasets.sh
