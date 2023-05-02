from redis import Redis
import os
from trueskill import Rating
from rocket_learn.rollout_generator.redis.utils import get_rating, _unserialize, _serialize, get_ratings, decode_buffers, PRETRAINED_QUALITIES


host = "127.0.0.1"
r = Redis(host=host,
          username="user1",
          password=os.environ["redis_user1_key"],
          db=4,)

# past_version_ratings = get_ratings('3v3', r)
# past_version_ratings = {k: v for k, v in sorted(past_version_ratings.items(), key=lambda item: item[1].mu)}
# quality_key = PRETRAINED_QUALITIES.format("3v3")
# nexto = {k.decode("utf-8"): Rating(*_unserialize(v)) for k, v in r.hgetall(quality_key).items()}

# new = Rating(mu=110, sigma=30)

for gamemode in ['1v1', '2v2', '3v3']:
    ratings = {k.decode("utf-8"): Rating(*_unserialize(v)) for k, v in r.hgetall(PRETRAINED_QUALITIES.format(gamemode)).items()}
    rating = ratings['Nexto']
    rating = Rating(rating.mu, 20)
    r.hset(PRETRAINED_QUALITIES.format(gamemode), key='Nexto', value=_serialize(tuple(rating)))

# for gamemode in ['1v1', '2v2', '3v3']:
#     r.hset(PRETRAINED_QUALITIES.format(gamemode), key='KBB', value=_serialize(tuple(new)))
#
# quality_key = PRETRAINED_QUALITIES.format("1v1")
# x = {k.decode("utf-8"): Rating(*_unserialize(v)) for k, v in r.hgetall(quality_key).items()}
# list(past_version_ratings)[-1]
