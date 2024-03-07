provider "aws" {
  region = "us-east-2"
}

locals {
  # common tags applied to all resources
  common_tags = {
    Project = "11711.rag"
  }
}

resource "aws_vpc" "test-env" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = local.common_tags
}

resource "aws_subnet" "subnet-uno" {
  cidr_block        = cidrsubnet(aws_vpc.test-env.cidr_block, 3, 1)
  vpc_id            = aws_vpc.test-env.id
  availability_zone = "us-east-2b"
  tags              = local.common_tags
}

resource "aws_security_group" "ingress-ssh-test" {
  name   = "allow-ssh-sg"
  vpc_id = aws_vpc.test-env.id
  tags   = local.common_tags

  ingress {
    cidr_blocks = ["0.0.0.0/0"]
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ingress-http-test" {
  name   = "allow-http-sg"
  vpc_id = aws_vpc.test-env.id
  tags   = local.common_tags

  ingress {
    cidr_blocks = ["0.0.0.0/0"]
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ingress-https-test" {
  name   = "allow-https-sg"
  vpc_id = aws_vpc.test-env.id
  tags   = local.common_tags

  ingress {
    cidr_blocks = ["0.0.0.0/0"]
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_eip" "ip-test-env" {
  instance = aws_spot_instance_request.test_worker.spot_instance_id
  vpc      = true
  tags     = local.common_tags
}

resource "aws_internet_gateway" "test-env-gw" {
  vpc_id = aws_vpc.test-env.id
  tags   = local.common_tags
}

resource "aws_route_table" "route-table-test-env" {
  vpc_id = aws_vpc.test-env.id
  tags   = local.common_tags

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.test-env-gw.id
  }
}

resource "aws_route_table_association" "subnet-association" {
  subnet_id      = aws_subnet.subnet-uno.id
  route_table_id = aws_route_table.route-table-test-env.id
}

resource "aws_key_pair" "rag_project_key" {
  key_name   = "rag_project_key"
  public_key = file("~/.ssh/id_rsa.pub")
}

resource "aws_spot_instance_request" "test_worker" {
  ami                    = "ami-02b696c88aad79a70"
  spot_price             = "1.2"
  instance_type          = "p3.2xlarge"
  spot_type              = "one-time"
  block_duration_minutes = "120"
  wait_for_fulfillment   = "true"
  key_name               = aws_key_pair.rag_project_key.key_name
  availability_zone      = "us-east-2b"
  tags                   = local.common_tags

  security_groups = [
    aws_security_group.ingress-ssh-test.id, aws_security_group.ingress-http-test.id,
    aws_security_group.ingress-https-test.id
  ]
  subnet_id = aws_subnet.subnet-uno.id
}

resource "aws_ebs_volume" "ebs_vol" {
  availability_zone = "us-east-2b"
  size              = 100
  type              = "gp2"
  tags              = local.common_tags
}

resource "aws_volume_attachment" "ebs_att" {
  device_name = "/dev/sdh"
  volume_id   = aws_ebs_volume.ebs_vol.id
  instance_id = aws_spot_instance_request.test_worker.spot_instance_id
}
