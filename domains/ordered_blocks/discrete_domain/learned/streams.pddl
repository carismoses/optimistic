(define (stream ordered-blocks-learned)
   (:predicate (TrustModel ?bt ?bb)
     (and (Block ?bt) (Block ?bb)))
)
