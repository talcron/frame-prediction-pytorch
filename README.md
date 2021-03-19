# WGAN Video Generation

## Dataset examples

### Tinyvideo Golf

![Golf](https://s3.amazonaws.com/comet.ml/image_5e54724b7ab941b7a158c5f0bea92cb0-vkEmHjctP4OpRWcfrtrkunjvR.gif)
![Golf](https://s3.amazonaws.com/comet.ml/image_5e54724b7ab941b7a158c5f0bea92cb0-mWNh6ZwMCh7W2VdUTJYVEqubv.gif)

### Tinyvideo Beach

![Beach](https://s3.amazonaws.com/comet.ml/image_82b88623474f4e32a9f2342b76774721-FmWDe0kEBKlbglTnCC8PNpZhx.gif)
![Beach](https://s3.amazonaws.com/comet.ml/image_82b88623474f4e32a9f2342b76774721-4V4wLOnXJpJlNW2WtLwN0XUon.gif)

### UCF-101 Golf

![UCF-Golf](https://s3.amazonaws.com/comet.ml/image_178c1c550180493586a58b69ef6ee8ca-TQa6U6moKnCdqK9YywS5RORDQ.gif)
![UCF-Golf](https://s3.amazonaws.com/comet.ml/image_178c1c550180493586a58b69ef6ee8ca-zW8fzGuSmBwP7aBYGgLZA8jME.gif)

## Results

### Tinyvideo

#### WGAN-GP

![Golf WGAN-GP](https://s3.amazonaws.com/comet.ml/image_8b264e0b617e4360bf4d5b1ba0f0e392-oklkt9X9kURthprfkOVY8NTqg.gif)
![Beach WGAN-GP](https://s3.amazonaws.com/comet.ml/image_af55525911124742a08bc13a96869d3c-ij3iJycy6BuDQRrdV4HalsOCB.gif)

#### SN

![Golf WGAN-SN](https://s3.amazonaws.com/comet.ml/image_5ccc418e6884417996bcb8a7a62474c1-ewIlvpjtHgqvhwVJhS81jzMaL.gif)
![Beach WGAN-SN](https://s3.amazonaws.com/comet.ml/image_c3a0d6dcd8d7435ca96846c156e84842-TpCHW0jRBu723QC5ciL3JT2GC.gif)

#### SN-GP

Due to poor results on the beach dataset, this was not repeated on the golf data.

![Beach WGAN-GP-SN](https://s3.amazonaws.com/comet.ml/image_82b88623474f4e32a9f2342b76774721-Fu4skbZKFIcFwVNrtqOrU5ijV.gif)

#### 0-GP

![Golf WGAN-0-GP](../media/zero_centered_fake_1_68k.gif?raw=true)

#### One-sided GP

![Golf One-sided-GP](https://s3.amazonaws.com/comet.ml/image_59e20d6b30d7473aab5036b23f277438-ikf9bG0nCak3qLY4wkROcKFW3.gif)

#### Drift Penalty GP

![Golf Drift Penalty GP](https://s3.amazonaws.com/comet.ml/image_5e54724b7ab941b7a158c5f0bea92cb0-N6Cg9UExKUWjKxDwD6o5Yxr9O.gif)
