FROM node:18
# Install Python
RUN apt-get update 
RUN apt-get install -y python3
RUN apt-get install -y python3-full 
RUN apt-get install -y python3-pip

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install dependencies
RUN npm install

RUN python3 --version
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-torch
RUN apt-get install -y python3-matplotlib
RUN apt-get install -y python3-pillow
RUN apt-get install -y python3-torchvision




# Expose the port the app runs on
EXPOSE 8080

# Start the application
CMD ["node", "api.js"]
