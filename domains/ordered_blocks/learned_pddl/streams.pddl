(define (stream ordered-blocks-learned)
  (:stream get-trust-model
    :inputs (?bt ?bb)
    :domain (and (Block ?bt) (Block ?bb))
    :fluents (On Clear OnTable)
    :outputs ()
    :certified (TrustModel ?bt ?bb)
  )
)
