# Cardinality-Estimation
Suppose you have a very large dataset - far too large to hold in memory - with duplicate entries. You want to know how many duplicate entries, but your data isn't sorted, and it's big enough that sorting and counting is impractical. How do you estimate how many unique entries the dataset contains? It's easy to see how this could be useful in many applications, such as query planning in a database: the best query plan can depend greatly on not just how many values there are in total, but also on how many unique values there are.

I'd encourage you to give this a bit of thought before reading onwards, because the algorithms we'll discuss today are quite innovative - and while simple, they're far from obvious.

A simple and intuitive cardinality estimator
Let's launch straight in with a simple example. Suppose someone generate a dataset with the following procedure:

Generate n evenly distributed random numbers
Arbitrarily replicate some of those numbers an unspecified number of times
Shuffle the resulting set of numbers arbitrarily
How can we estimate how many unique numbers there are in the resulting dataset? Knowing that the original set of numbers was random and evenly distributed, one very simple possibility occurs: simply find the smallest number in the set. If the maximum possible value is m, and the smallest value we find is x, we can then estimate there to be about m/x unique values in the total set. For instance, if we scan a dataset of numbers between 0 and 1, and find that the smallest value in the set is 0.01, it's reasonable to assume there are roughly 100 unique values in the set; any more and we would expect to see a smaller minimum value. Note that it doesn't matter how many times each value is repeated: it is the nature of aggregates like min that repetitions do not affect the output value.

This procedure has the advantage of being extremely straightforward, but it's also very inaccurate. It's not hard to imagine a set with only a few distinct values containing an unusually small number; likewise a set with many distinct values could have a smallest value that is larger than we expect. Finally, few datasets are so well behaved as to be neatly random and evenly distributed. Still, this proto-algorithm gives us some insight into one possible approach to get what we want; what we need is further refinements.

Probabilistic counting
The first set of refinements comes from the paper Probabilistic Counting Algorithms for Data Base Applications by Flajolet and Martin, with further refinements in the papers LogLog counting of large cardinalities by Durand-Flajolet, and HyperLogLog: The analysis of a near-optimal cardinality estimation algorithm by Flajolet et al. It's interesting to watch the development and improvement of the ideas from paper to paper, but I'm going to take a slightly different approach and demonstrate how to build and improve a solution from the ground up, omitting some of the algorithm from the original paper. Interested readers are advised to read through all three; they contain a lot of mathematical insights I won't go into in detail here.

First, Flajolet and Martin observe that given a good hash function, we can take any arbitrary set of data and turn it into one of the sort we need, with evenly distributed, (pseudo-)random values. With this simple insight, we can apply our earlier procedure to whatever data we want, but they're far from done.

Next, they observe that there are other patterns we can use to estimate the number of unique values, and some of them perform better than recording the minimum value of the hashed elements. The metric Flajolet and Martin pick is counting the number of 0 bits at the beginning of the hashed values. It's easy to see that in random data, a sequence of k zero bits will occur once in every 2k elements, on average; all we need to do is look for these sequences and record the length of the longest sequence to estimate the total number of unique elements. This still isn't a great estimator, though - at best it can give us a power of two estimate of the number of elements, and much like the min-value based estimate, it's going to have a huge variance. On the plus side, our estimate is very small: to record sequences of leading 0s of up to 32 bits, we only need a 5 bit number.

As a side note, the original Flajolet-Martin paper deviates here and uses a bitmap-based procedure to get a more accurate estimate from a single value. I won't go into this in detail, since it's soon obsoleted by improvements in subsequent papers; interested readers can read the original paper for more details.

So we now have a rather poor estimate of the number of values in the dataset based on bit patterns. How can we improve on it? One straightforward idea is to use multiple independent hash functions. If each hash produces its own set of random outputs, we can record the longest observed sequence of leading 0s from each; at the end we can average our values for a more accurate estimate.

This actually gives us a pretty good result statistically speaking, but hashing is expensive. A better approach is one known as stochastic averaging. Instead of using multiple hash functions, we use just a single hash function, but use part of its output to split values into one of many buckets. Supposing we want 1024 values, we can take the first 10 bits of the hash function as a bucket number, and use the remainder of the hash to count leading 0s. This loses us nothing in terms of accuracy, but saves us a lot of redundant computation of hashes.

Applying what we've learned so far, here's a simple implementation. This is equivalent to the LogLog algorithm in the Durand-Flajolet paper; for convenience and clarity, though, I'm counting trailing (least-significant) 0 bits rather than leading ones; the result is exactly equivalent.

def trailing_zeroes(num):
  """Counts the number of trailing 0 bits in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

def estimate_cardinality(values, k):
  """Estimates the number of unique elements in the input set values.

  Arguments:
    values: An iterator of hashable elements to estimate the cardinality of.
    k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
  """
  num_buckets = 2 ** k
  max_zeroes = [0] * num_buckets
  for value in values:
    h = hash(value)
    bucket = h & (num_buckets - 1) # Mask out the k least significant bits as bucket ID
    bucket_hash = h >> k
    max_zeroes[bucket] = max(max_zeroes[bucket], trailing_zeroes(bucket_hash))
  return 2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402
This is all pretty much as we just described: we keep a bunch of counts of number of leading (or trailing) zeroes; at the end we average the counts; if our average is x, our estimate is 2x, multiplied by the number of buckets. Not mentioned previously is this magic number 0.79402. Statistical analysis shows that our procedure introduces a predictable bias towards larger estimates; this magic constant is derived in the paper by Durand-Flajolet to correct that bias. The actual figure varies with the number of buckets used, but with larger numbers of buckets (at least 64), it converges on the estimate we use in the above algorithm. See the complete paper for lots more information, including the derivation of that number.

This procedure gives us a pretty good estimate - for m buckets, the average error is about 1.3/sqrt(m). Thus with 1024 buckets (for 1024 * 5 = 5120 bits, or 640 bytes), we can expect an average error of about 4%; 5 bits per bucket is enough to estimate cardinalities up to 227 per the paper). That's pretty good for less than a kilobyte of memory!

Let's try it ourselves on some random data:

>>> [100000/estimate_cardinality([random.random() for i in range(100000)], 10) for j in range(10)]
[0.9825616152548807, 0.9905752876839672, 0.979241749110407, 1.050662616357679, 0.937090578752079, 0.9878968276629505, 0.9812323203117748, 1.0456960262467019, 0.9415413413873975, 0.9608567203911741]
Not bad! Some of the estimates are off by more than the predicted 4%, but all in all they're pretty good. If you're trying this experiment yourself, one caution: Python's builtin hash() hashes integers to themselves. As a result, running something like estimate_cardinality(range(10000), 10) will give wildly divergent results, because hash() isn't behaving like a good hash function should. Using random numbers as in the example above works just fine, however.

Improving accuracy: SuperLogLog and HyperLogLog
While we've got an estimate that's already pretty good, it's possible to get a lot better. Durand and Flajolet make the observation that outlying values do a lot to decrease the accuracy of the estimate; by throwing out the largest values before averaging, accuracy can be improved. Specifically, by throwing out the 30% of buckets with the largest values, and averaging only 70% of buckets with the smaller values, accuracy can be improved from 1.30/sqrt(m) to only 1.05/sqrt(m)! That means that our earlier example, with 640 bytes of state and an average error of 4% now has an average error of about 3.2%, with no additional increase in space required.

Finally, the major contribution of Flajolet et al in the HyperLogLog paper is to use a different type of averaging, taking the harmonic mean instead of the geometric mean we just applied. By doing this, they're able to edge down the error to 1.04/sqrt(m), again with no increase in state required. The complete algorithm is somewhat more complicated, however, as it requires corrections for both small and large cardinalities. Interested readers should - you guessed it - read the entire paper for details.

Parallelization
One really neat attribute that all these schemes share is that they're really easy to parallelize. Multiple machines can independently run the algorithm with the same hash function and the same number of buckets; at the end results can be combined by taking the maximum value of each bucket from each instance of the algorithm. Not only is this trivial to do, but the resulting estimate is exactly identical to the result we'd get running it on a single machine, while we only needed to transfer less than a kilobyte of data per instance to achieve this.

Conclusion
Cardinality estimation algorithms like the ones we've just discussed make it possible to get a very good estimate - within a few percent - of the total number of unique values in a dataset, typically using less than a kilobyte of state. We can do this regardless of the nature of the data, and the work can be distributed over multiple machines with minimum coordination overhead and data transfer. The resulting estimates can be useful for a range of things, such as traffic monitoring (how many unique IPs is a host contacting?) and database query optimization (should we sort and merge, or construct a hashtable of unique values?).
